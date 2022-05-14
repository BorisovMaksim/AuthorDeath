
cols_to_drop = [
    # 'const', 'square_m', 'ExhibitedNum', 'ProvenanceNum', 'LiteratureNum', 'CataloguingLength',
    'date_of_birth',
    #         'date_of_death',
    'century',
    #         'normalized_price', 'Technique_acrylic',
    #         'Technique_aquatint', 'Technique_casein', 'Technique_chalk', 'Technique_charcoal',
    #         'Technique_crayon', 'Technique_drypoint', 'Technique_enamel', 'Technique_engraving',
    #         'Technique_etching',
    'Technique_gloss',
    #         'Technique_gouache', 'Technique_graphite',
    #         'Technique_ink',
    'Technique_lithograph',
    #         'Technique_mixed media',
    'Technique_oil',
    #         'Technique_pastel', 'Technique_pen',
    'Technique_pencil',
    #         'Technique_pigment',
    'Technique_print',
    #         'Technique_serigraph', 'Technique_tempera', 'Technique_watercolor',
    'Technique_woodcut', 'Technique_woven',
    'Technique_None',
    # 'Material_board', 'Material_canvas',
    # 'Material_card',
    'Material_gold',
    'Material_lamina',
    # 'Material_linen',
    'Material_linoleum',
    'Material_masonite',
    # 'Material_panel',
    'Material_paper',
    'Material_serigraph',
    'Material_silk',
    # 'Material_tapestry',
    'Material_tin',
    # 'Material_toile', 'Material_volant', 'Material_watercolor',
    # 'Material_wood',
    'Material_None',
    'Currency_CHF',
    'Currency_CNY',
    # 'Currency_ESP', 'Currency_EUR',
    'Currency_GBP',
    'Currency_HKD',
    #         'Currency_RMB', 'Currency_USD',
    'nationality_Afro-American',
    #         'nationality_American', 'nationality_Austrian',
    'nationality_Belgian',
    #         'nationality_Catalonian',
    'nationality_Chinese',
    #         'nationality_Dutch', 'nationality_English', 'nationality_Fleming',
    'nationality_Florentine',
    'nationality_French',
    #         'nationality_German',
    'nationality_Greek',
    #         'nationality_Italian', 'nationality_Japanese', 'nationality_Jew', 'nationality_Netherlander',
    #         'nationality_Norwegian', 'nationality_Polish',
    'nationality_Romanian',
    #         'nationality_Russian',
    'nationality_Scottish',
    'nationality_Spaniard',
    # 'nationality_Swiss',
    # 'sex_F',
    'sex_M',
    # 'style_abstract expressionism', 'style_abstractionism', 'style_academicism', 'style_avant-garde',
    # 'style_baroque', 'style_classicism', 'style_conceptualism',
    'style_contemporary art',
    # 'style_cubism',
    # 'style_dadaism',
    'style_expressionism',
    # 'style_fauvism',
    'style_folk-art',
    # 'style_futurism',
    # 'style_impressionism',
    'style_installation',
    # 'style_magical realism',
    'style_mannerism',
    # 'style_metaphysical art', 'style_minimalism',
    # 'style_modern',
    'style_modernism',
    # 'style_nabism',
    'style_neo-expressionism',
    # 'style_neo-impressionism',
    # 'style_neo-pop / post-pop',
    # 'style_neoclassicism', 'style_organic abstraction',
    'style_pop-art',
    # 'style_post-impressionism',
    # 'style_postconceptualism', 'style_pre-raphaelitism',
    'style_realism',
    # 'style_renaissance',
    'style_rococo',
    'style_romanica',
    # 'style_romanticism',
    'style_surrealism',
    # 'style_symbolism',
    # 'style_traditional chinese painting',

    # 'feature1_coeval', 'feature1_communist',
    #         'feature1_died in a car accident', 'feature1_earl',
    'feature1_friar',
    #         'feature1_homosexual',
    #         'feature1_killed on war', 'feature1_killer', 'feature1_migrant',
    'feature1_missionary',
    #         'feature1_philatelist',
    'feature1_philosopher',
    #         'feature1_royal descent',
    'feature1_scientist',
    'feature1_sculptor', 'feature1_self-taught', 'feature1_suicide', 'feature1_writer',
    'feature1_�',
    'feature1_None', 'feature2_died in a car accident', 'feature2_director',
    'feature2_disabled',
    'feature2_homosexual',
    'feature2_migrant', 'feature2_scenographer',
    'feature2_scientist',
    'feature2_sculptor', 'feature2_social activist', 'feature2_�', 'feature2_None',
    #         "Auction_Christie's", 'Auction_Phillips',
    "Auction_Sotheby's"
]