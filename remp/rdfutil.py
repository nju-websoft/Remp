import itertools

import rdflib
import rdflib.store
import networkx as nx
import pandas as pd
import numpy as np

namespace_inverse_index = {}
for k, v in itertools.chain(rdflib.__dict__.items(), rdflib.namespace.__dict__.items()):
    if type(v) == rdflib.namespace.Namespace or type(v) == rdflib.namespace.ClosedNamespace:
        namespace_inverse_index[str(v)[0:-1]] = k.lower()

namespace_inverse_index['http://www.w3.org/1999/02/22-rdf-syntax-ns'] = 'rdfs'
namespace_inverse_index['http://www.instancematching.org/IIMB2012/ADDONS'] = 'addons'
namespace_inverse_index['http://www.w3.org/2003/01/geo/wgs84_pos'] = 'wgs84_pos'
namespace_inverse_index['http://kmi.open.ac.uk/fusion/dblp'] = 'dblp'
namespace_inverse_index['http://lsdis.cs.uga.edu/projects/semdis/opus'] = 'opus'

namespace_prefixes = [
    ('http://purl.org/dc/elements/1.1/', 'purl'),
    ('http://oaei.ontologymatching.org/2012/IIMBDATA/en/', 'iimb'),
    ('http://oaei.ontologymatching.org/2012/IIMBTBOX/', 'iimbtbox'),
    ('http://oaei.ontologymatching.org/2012/IIMBDATA/', 'iimbdata'),
    ('http://www.instancematching.org/IIMB2012/', 'iimb2012'),
    ('http://yago-knowledge.org/resource/', 'yago'),
    ('http://dbpedia.org/resource/', 'dbp'),
    ('http://dbpedia.org/ontology/', 'dbo'),
    ('http://xmlns.com/foaf/0.1/', 'foaf'),
    ('http://purl.org/ontology', 'purl'),
    ('http://schema.org', 'schema'),
    ('http://www.georss.org/georss/', 'georss'),
    ('http://dblp.uni-trier.de/rec/bibtex/', 'bibtex'),
    ('http://www.informatik.uni-trier.de/~ley/db/', 'db')
]


def shrink_uri(uri):
    if type(uri) != rdflib.term.URIRef and type(uri) != str:
        return uri
    parts = str(uri).split('#')
    if len(parts) == 2 and parts[0] in namespace_inverse_index:
        return namespace_inverse_index[parts[0]] + ':' + parts[1]
    else:
        for (prefix, ns) in namespace_prefixes:
            if uri.startswith(prefix):
                return ns + ':' + uri[len(prefix):]
#         print('Warning: %s does not have namespace prefix' % uri)
        return uri


class TripleIteratorStore(rdflib.store.Store):
    formula_aware = True
    context_aware = True
    graph_aware = False

    def __init__(self, attribute_handler, relation_handler, bnode_handler):
        rdflib.store.Store.__init__(self)
        self.attribute_handler = attribute_handler
        self.relation_handler = relation_handler
        self.bnode_handler = bnode_handler

    def add(self, triple, context, quoted=False):
        if type(triple[0]) == rdflib.term.BNode or type(triple[2]) == rdflib.term.BNode:
            if self.bnode_handler is not None:
                self.bnode_handler(triple)
        elif type(triple[2]) == rdflib.term.URIRef:
            self.relation_handler(triple)
        elif type(triple[2]) == rdflib.term.Literal:
            self.attribute_handler(triple)

    def remove(self, triple, context=None):
        if triple == (None, None, None):
            return

    def bind(self, prefix, namespace):
        pass


OAEI_ALIGNMENT_NAMESPACE = rdflib.namespace.Namespace('http://knowledgeweb.semanticweb.org/heterogeneity/alignment#')


def read_oaei_alignment(rdf_file_path):
    r = rdflib.Graph()
    r.parse(source=rdf_file_path)

    align = r.query(
        'select ?e1 ?e2 where { ?map oaei:entity1 ?e1 . ?map oaei:entity2 ?e2 }',
        initNs={'oaei': OAEI_ALIGNMENT_NAMESPACE}
    )

    return [(shrink_uri(x), shrink_uri(y)) for (x, y) in align]


class TripleLoader(object):
    def __init__(self):
        self.relations = []
        self.attributes = []
        self.bnodes = []
        store = TripleIteratorStore(
            self._handle_attribute_triple,
            self._handle_relation_triple,
            self._handle_bnode_triple
        )
        self.graph = rdflib.ConjunctiveGraph(store)

    def _handle_attribute_triple(self, triple):
        (s, a, v) = triple
        self.attributes.append(dict(s=shrink_uri(s), a=shrink_uri(a), v=v))

    def _handle_relation_triple(self, triple):
        (s, r, o) = triple
        self.relations.append(dict(s=shrink_uri(s), r=shrink_uri(r), o=shrink_uri(o)))

    def _handle_bnode_triple(self, triple):
        (s, p, o) = triple
        if type(s) == rdflib.term.URIRef:
            s = shrink_uri(s)
        if type(o) == rdflib.term.URIRef:
            o = shrink_uri(o)
        self.bnodes.append(dict(s=s, p=shrink_uri(p), o=o))

    def load(self, *args, **kwargs):
        self.graph.parse(*args, **kwargs)


def extract_axioms(bnodes, relations):
    axioms = pd.DataFrame(bnodes, columns=['s', 'p', 'o'])
    disjoint_types = []
    for disjoint_axiom in axioms[axioms['o'] == 'owl:AllDisjointClasses']['s']:
        classes = []
        next_member = axioms[(axioms['s'] == disjoint_axiom) & (axioms['p'] == 'owl:members')]['o'].values
        while len(next_member) > 0 and type(next_member) == np.ndarray:
            next_member = next_member[0]
            classes.append(axioms[(axioms['s'] == next_member) & (axioms['p'] == 'rdfs:first')]['o'].values)
            next_member = axioms[(axioms['s'] == next_member) & (axioms['p'] == 'rdfs:rest')]['o'].values
        classes = [c[0] for c in classes[:-1]]
        disjoint_types.append(classes)
    
    inverse_properties = relations[relations['r'] == 'owl:inverseOf'][['s', 'o']].values
    
    sub_classes = relations[relations['r'] == 'rdfs:subClassOf']
    class_graph = nx.from_pandas_edgelist(sub_classes, source='o', target='s', create_using=nx.DiGraph())
    return dict(disjoint_types=disjoint_types, inverse_properties=inverse_properties, type_graph=class_graph)


def remove_ontology(relation_dataframe, relation='r'):
    relation_dataframe = relation_dataframe[relation_dataframe[relation] != 'owl:inverseOf']
    relation_dataframe = relation_dataframe[relation_dataframe[relation] != 'rdfs:type']
    relation_dataframe = relation_dataframe[relation_dataframe[relation] != 'rdfs:domain']
    relation_dataframe = relation_dataframe[relation_dataframe[relation] != 'rdfs:range']
    relation_dataframe = relation_dataframe[relation_dataframe[relation] != 'rdfs:subClassOf']
    relation_dataframe = relation_dataframe[relation_dataframe[relation] != 'rdfs:subPropertyOf']
    return relation_dataframe